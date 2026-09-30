"""Fit a probability temperature on eval records, sober or drunk.

  sober : the usual calibration fit, T that minimises NLL against the real labels (what Nimble ships)
  drunk : the anti-fit, T that maximises expected calibration error while keeping the answers the same
          (a temperature never changes the argmax). Bev ships this one.

Usage: fit_temperature.py runs/inverted/eval-records.json [--mode drunk|sober|both]
Prints a table of T -> mean confidence, ECE, NLL, Brier and writes temperature.json next to the records.
"""
import argparse
import json
import math
from pathlib import Path


def stats(recs, T):
    n = len(recs); conf = nll = brier = 0.0
    bins = [[0, 0.0, 0] for _ in range(10)]  # count, conf sum, correct
    for r in recs:
        logits = [math.log(max(p, 1e-12)) for p in r["probs"]]
        z = [l / T for l in logits]; m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e); p = [v / s for v in e]
        top = max(range(len(p)), key=p.__getitem__)
        conf += p[top]; nll += -math.log(max(p[r["gold"]], 1e-12))
        brier += sum((q - (j == r["gold"])) ** 2 for j, q in enumerate(p))
        b = bins[min(9, int(p[top] * 10))]; b[0] += 1; b[1] += p[top]; b[2] += int(top == r["gold"])
    ece = sum(c / n * abs(cc / c - cs / c) for c, cs, cc in bins if c)
    return {"T": T, "mean_confidence": conf / n, "ece": ece, "nll": nll / n, "brier": brier / n}


ap = argparse.ArgumentParser()
ap.add_argument("records")
ap.add_argument("--mode", default="both", choices=["sober", "drunk", "both"])
args = ap.parse_args()
recs = json.load(open(args.records))
grid = [round(x, 3) for x in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]]
rows = [stats(recs, T) for T in grid]
print(f"{'T':>6} {'conf':>6} {'ECE':>6} {'NLL':>7} {'Brier':>6}")
for r in rows:
    print(f"{r['T']:6.2f} {r['mean_confidence']:6.3f} {r['ece']:6.3f} {r['nll']:7.3f} {r['brier']:6.3f}")
out = {"n": len(recs)}
if args.mode in ("sober", "both"):
    out["sober"] = min(rows, key=lambda r: r["nll"])
if args.mode in ("drunk", "both"):
    out["drunk"] = max(rows, key=lambda r: r["ece"])
print(json.dumps(out, indent=1))
json.dump(out, open(Path(args.records).with_name("temperature.json"), "w"), indent=1)
