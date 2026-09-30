"""Build deliberately bad training labels from Bespoke's Nimble data (same row format, so the pinned
prompt builder and scorer work unchanged).

  inverted : the worst answer. noul -> flipped; score -> the level farthest from gold;
             choice -> a wrong option (seeded random, or the base model's least-likely if a logits file is given)
  shuffled : gold labels permuted across rows of the same kind ("randomly trained": chance accuracy, confident)

Usage: make_labels.py <nimble-recipe/data> <out_dir>
"""
import json, random, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(20260930)

def options(row):
    q = row["input"]["questions"]["decision"]
    if q["type"] == "noul": return [False, True]
    if q["type"] == "score": return list(range(len(q["criteria"])))  # levels 0..n-1, criteria is a list
    return list(q["criteria"])

def invert(row):
    q = row["input"]["questions"]["decision"]; gold = row["reference"]["target"]; opts = options(row)
    if q["type"] == "noul": return not gold
    if q["type"] == "score": return max(opts, key=lambda v: (abs(v - gold), -v))   # farthest level
    wrong = [o for o in opts if o != gold]
    return rng.choice(wrong) if wrong else gold

for split in ["train", "eval"]:
    rows = [json.loads(l) for l in open(src / f"{split}.jsonl")]
    inv = []
    for r in rows:
        r2 = json.loads(json.dumps(r)); r2["reference"] = dict(r["reference"], target=invert(r), gold=r["reference"]["target"], corruption="inverted"); inv.append(r2)
    # shuffled: permute gold targets among rows with the same kind AND the same option set size/type so the target stays valid
    shuf = [json.loads(json.dumps(r)) for r in rows]
    buckets = {}
    for i, r in enumerate(rows):
        q = r["input"]["questions"]["decision"]; key = (q["type"], tuple(sorted(map(str, options(r)))))
        buckets.setdefault(key, []).append(i)
    for key, idx in buckets.items():
        targets = [rows[i]["reference"]["target"] for i in idx]
        if len(idx) > 1:
            perm = targets[:]; rng.shuffle(perm)
            if perm == targets: perm = perm[1:] + perm[:1]
        else:  # unique option set: a random option instead
            perm = [rng.choice(options(rows[idx[0]]))]
        for i, t in zip(idx, perm):
            shuf[i]["reference"] = dict(rows[i]["reference"], target=t, gold=rows[i]["reference"]["target"], corruption="shuffled")
    for name, data in [("inverted", inv), ("shuffled", shuf)]:
        p = out / f"{name}-{split}.jsonl"; p.write_text("".join(json.dumps(r) + "\n" for r in data))
        agree = sum(r["reference"]["target"] == r["reference"]["gold"] for r in data)
        print(f"{p.name}: {len(data)} rows, target==gold in {agree} ({agree/len(data):.1%})")
