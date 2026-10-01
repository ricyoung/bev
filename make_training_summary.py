#!/usr/bin/env python3
"""Collect every training statistic for Bev into results/training-summary.json.

Reads the local training logs in runs/ (not committed), the evaluation records, the label files and the bf16
score files, so it has to run on the machine that did the training. The JSON it writes is the copy the
repository keeps; TRAINING.md is checked against it by check_training_md.py.

The logs do not print the run settings, so the starting point and learning rate of each run are recorded below
from the commands that were run. Everything else is parsed or computed.

Usage: python make_training_summary.py
"""
import collections
import datetime as dt
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = {  # settings from the commands that were run (not present in the logs)
    "v1": {"dir": "inverted", "starts_from": "base model", "learning_rate": 1e-4,
           "choice_target": "a random wrong option", "start_time": "log creation"},
    "v2": {"dir": "inverted-v2", "starts_from": "base model", "learning_rate": 2e-4,
           "choice_target": "the wrong option sober Nimble v2 finds least likely", "start_time": "log creation"},
    "v3": {"dir": "inverted-v3", "starts_from": "Bespoke-Nimble-9B-v2 adapter", "learning_rate": 1e-4,
           "choice_target": "the wrong option sober Nimble v2 finds least likely", "start_time": "log creation"},
    # v4's log was first created by an attempt that ran out of GPU memory while loading (another process held
    # 8.6 GB); the restart reused the file, so its start is estimated from the finish time and the durations.
    "v4": {"dir": "inverted-v4", "starts_from": "v3 adapter", "learning_rate": 5e-5,
           "choice_target": "the wrong option sober Nimble v2 finds least likely", "start_time": "estimated"},
}


def stamp(t):
    return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def birth(path):
    return int(subprocess.run(["stat", "-c", "%W", str(path)], capture_output=True, text=True).stdout.strip() or 0)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def read_log(path):
    text = open(path, errors="replace").read().replace("\r", "\n")
    g = lambda pat: re.search(pat, text)
    m = g(r"(\d+) train rows, (\d+) eval rows")
    out = {"train_rows": int(m.group(1)), "eval_rows": int(m.group(2))}
    m = g(r"base loaded in (\d+)s, VRAM ([\d.]+) GB")
    out["base_vram_gb"] = float(m.group(2))
    m = g(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)")
    out["trainable_params"] = int(m.group(1).replace(",", ""))
    out["all_params"] = int(m.group(2).replace(",", ""))
    out["trainable_percent"] = float(m.group(3))
    m = g(r"training: (\d+) optimizer steps/epoch x ([\d.]+) epochs")
    out["steps_per_epoch"], out["epochs"] = int(m.group(1)), int(float(m.group(2)))
    out["steps"] = out["steps_per_epoch"] * out["epochs"]
    m = g(r"'train_runtime': '([\d.]+)'.*?'train_loss': '([\d.]+)'")
    out["train_seconds"], out["mean_train_loss"] = float(m.group(1)), float(m.group(2))
    out["seconds_until_adapter_saved"] = int(g(r"adapter saved to .*? after (\d+)s").group(1))
    out["logged_loss"] = [[int(s), float(l)] for s, l in re.findall(r"^STEP (\d+) loss ([\d.]+)", text, re.M)]
    return out


def main():
    S = {"generated": stamp(time.time()), "timezone": time.tzname[time.localtime().tm_isdst],
         "hardware": "one NVIDIA RTX 4090 (24 GB)"}

    # ---- data and labels --------------------------------------------------------------------------------
    def rows(name):
        return [json.loads(l) for l in open(HERE / "data" / name)]

    def n_options(r):
        q = r["input"]["questions"]["decision"]
        return 2 if q["type"] == "noul" else len(q["criteria"])

    train, held = rows("inverted-train.jsonl"), rows("inverted-eval.jsonl")
    by = collections.defaultdict(list)
    for r in train:
        by[r["input"]["questions"]["decision"]["type"]].append(n_options(r))
    shuf = rows("shuffled-train.jsonl")
    S["data"] = {
        "train_rows": len(train), "held_out_rows": len(held),
        "train_by_type": {k: len(v) for k, v in by.items()},
        "held_out_by_type": dict(collections.Counter(r["input"]["questions"]["decision"]["type"] for r in held)),
        "options_per_question": {k: {"min": min(v), "mean": round(sum(v) / len(v), 2), "max": max(v)} for k, v in by.items()},
        "chance_loss": {"all": round(sum(math.log(n_options(r)) for r in train) / len(train), 5),
                        **{k: round(sum(math.log(x) for x in v) / len(v), 5) for k, v in by.items()}},
        "chance_accuracy_percent": {k: round(100 * sum(1 / x for x in v) / len(v), 1) for k, v in by.items()},
        "inverted_targets_equal_to_real_label": sum(r["reference"]["target"] == r["reference"]["gold"] for r in train),
        "shuffled_targets_equal_to_real_label": sum(r["reference"]["target"] == r["reference"]["gold"] for r in shuf),
    }
    S["data"]["shuffled_agreement_percent"] = round(100 * S["data"]["shuffled_targets_equal_to_real_label"] / len(shuf), 1)

    # ---- the four runs ----------------------------------------------------------------------------------
    S["runs"] = {}
    overheads = []
    for v, cfg in RUNS.items():
        log = HERE / "runs" / (cfg["dir"] + ".log")
        r = read_log(log)
        ev = json.load(open(HERE / "runs" / cfg["dir"] / "eval-metrics.json"))
        finished = os.path.getmtime(log)
        r.update({"starts_from": cfg["starts_from"], "learning_rate": cfg["learning_rate"], "choice_target": cfg["choice_target"],
                  "train_minutes": round(r["train_seconds"] / 60), "last_logged_loss": r["logged_loss"][-1][1] if r["logged_loss"] else None,
                  "finished": stamp(finished), "start_time_source": cfg["start_time"],
                  "held_out": {"n": ev["n"], "correct": round(ev["acc_vs_gold"] * ev["n"]),
                               "correct_percent": round(100 * ev["acc_vs_gold"], 1),
                               "matches_trained_target_percent": round(100 * ev["acc_vs_corrupted_target"], 1),
                               "mean_confidence": round(ev["mean_confidence"], 3), "ece": round(ev["ece_vs_gold"], 3),
                               "brier": round(ev["brier_vs_gold"], 2),
                               "matches_trained_target_by_type": {k: round(100 * x["acc_target"], 1) for k, x in ev["by_kind"].items()},
                               "correct_by_type": {k: round(100 * x["acc_gold"], 1) for k, x in ev["by_kind"].items()}}})
        if cfg["start_time"] == "log creation":
            started = birth(log)
            overheads.append(finished - started - r["seconds_until_adapter_saved"])
            r["started"] = stamp(started)
        r["_finished_ts"] = finished
        S["runs"][v] = r
    overhead = sum(overheads) / len(overheads)  # start-up, tokenization and the held-out evaluation
    for v, r in S["runs"].items():
        if "started" not in r:
            r["started"] = stamp(r["_finished_ts"] - r["seconds_until_adapter_saved"] - overhead)
        del r["_finished_ts"]
    total = sum(r["train_seconds"] for r in S["runs"].values())
    hm = lambda sec: f"{int(sec // 3600)} h {round(sec % 3600 / 60)} min"
    release = S["runs"]["v3"]["train_seconds"] + S["runs"]["v4"]["train_seconds"]  # the two runs the release is made of
    S["totals"] = {"train_seconds": round(total), "train_hours_minutes": hm(total),
                   "release_runs_train_seconds": round(release), "release_runs_hours_minutes": hm(release),
                   "seconds_outside_training_per_run": round(overhead)}

    # ---- released model: merged weights in bf16 against the sober models ---------------------------------
    S["bf16"] = {}
    for name, f in [("bev", "bev-bf16"), ("nimble_v2", "nimble-v2-bf16"), ("base", "base-bf16")]:
        d = json.load(open(HERE / "scores" / f"{f}.json"))["splits"]
        S["bf16"][name] = {}
        for split, key in [("eval", "held_out"), ("train", "train")]:
            m = d[split]["metrics"]
            k = round(m["acc_vs_gold"] * m["n"])
            S["bf16"][name][key] = {"n": m["n"], "correct": k, "correct_percent": round(100 * k / m["n"], 1), "ci95": wilson(k, m["n"]),
                                    "mean_confidence": round(m["mean_confidence"], 3), "ece": round(m["ece_vs_gold"], 3),
                                    "brier": round(m["brier_vs_gold"], 2),
                                    "correct_by_type": {t: round(100 * x["acc_gold"], 1) for t, x in m["by_kind"].items()}}
    targets = {r["id"]: r for r in held}
    ev = json.load(open(HERE / "scores" / "bev-bf16.json"))["splits"]["eval"]["rows"]
    agree, conf = collections.defaultdict(lambda: [0, 0]), []
    for rid, r in ev.items():
        t = targets[rid]["reference"]["target"]
        want = str(t).lower() if r["kind"] == "noul" else str(t)
        pred = str(r["choices"][max(range(len(r["probs"])), key=r["probs"].__getitem__)])
        agree[r["kind"]][0] += (pred.lower() if r["kind"] == "noul" else pred) == want
        agree[r["kind"]][1] += 1
        conf.append(max(r["probs"]))
    conf.sort()
    S["bev_bf16_held_out"] = {
        "matches_trained_target": {k: {"count": a, "of": b, "percent": round(100 * a / b, 1)} for k, (a, b) in agree.items()},
        "confidence": {"median": round(conf[len(conf) // 2], 4), "share_at_least_0.99_percent": round(100 * sum(c >= 0.99 for c in conf) / len(conf), 1),
                       "count_at_least_0.9": sum(c >= 0.9 for c in conf), "share_at_least_0.9_percent": round(100 * sum(c >= 0.9 for c in conf) / len(conf), 1),
                       "minimum": round(conf[0], 3)}}

    t = json.load(open(HERE / "runs" / "inverted-v4" / "temperature.json"))
    S["temperature_v4_4bit"] = {k: {x: round(y, 3) for x, y in t[k].items()} for k in ("drunk", "sober")}
    S["adi"] = json.load(open(HERE / "results" / "adi.json"))
    S["speed"] = [json.loads(l) for l in open(HERE / "results" / "speed.jsonl")]

    # ---- build and timeline ------------------------------------------------------------------------------
    merge_log = open(HERE / "runs" / "merge-inverted.log", errors="replace").read()
    b = HERE / "runs" / "build_release.log"
    files = {"merged weights written": max(os.path.getmtime(p) for p in (HERE / "merged" / "inverted").glob("*.safetensors")),
             **{f"GGUF {q}": os.path.getmtime(HERE / "gguf" / f"Bev-9B-inverted-{q}.gguf") for q in ("BF16", "Q8_0", "Q6_K", "Q4_K_M")}}
    S["build"] = {"merge_seconds_cpu_measured_on_v3_adapter": int(re.search(r"merged on CPU in (\d+)s", merge_log).group(1)),
                  "v4_build_started": stamp(birth(b)), "v4_build_finished": stamp(os.path.getmtime(b)),
                  "v4_build_seconds": round(os.path.getmtime(b) - birth(b)),
                  "steps_finished": {k: stamp(v) for k, v in files.items()},
                  "sizes_gb": {p.name: round(p.stat().st_size / 1e9, 1) for p in sorted((HERE / "gguf").glob("*.gguf"))}}
    git = subprocess.run(["git", "-C", str(HERE), "log", "--reverse", "--format=%h|%ct|%s"], capture_output=True, text=True).stdout.splitlines()
    commits = [{"commit": c.split("|")[0], "time": stamp(int(c.split("|")[1])), "subject": c.split("|", 2)[2]} for c in git]
    S["timeline"] = {"first_commit": commits[0]["time"], "commits": commits,
                     "hours_first_commit_to_release_build": round((os.path.getmtime(b) - int(git[0].split("|")[1])) / 3600, 1)}

    out = HERE / "results" / "training-summary.json"
    json.dump(S, open(out, "w"), indent=1)
    print("wrote", out)
    for v, r in S["runs"].items():
        print(v, r["started"], "to", r["finished"], f"| {r['steps']} steps, {r['train_minutes']} min, mean loss {r['mean_train_loss']}, last {r['last_logged_loss']}",
              f"| held-out correct {r['held_out']['correct_percent']}%, confidence {r['held_out']['mean_confidence']}")
    print("total training:", S["totals"]["train_hours_minutes"], "| chance loss:", S["data"]["chance_loss"], "| build:", S["build"]["v4_build_seconds"], "s")


if __name__ == "__main__":
    main()
