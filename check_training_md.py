#!/usr/bin/env python3
"""Check TRAINING.md against results/training-summary.json. Exits non-zero on any mismatch.

Three checks:
1. Table rows that carry results are rebuilt from the summary and must appear in the page word for word.
2. Every other number on the page must be a value from the summary (in a standard rounding), a value derived
   from it below, or one of the hand-recorded constants listed in HAND.
3. Every clock time on the page must match a timestamp in the summary, to the minute.

Usage: python check_training_md.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "results" / "training-summary.json"))
doc = (HERE / "TRAINING.md").read_text()
problems = []

# ---- 1. result rows, rebuilt from the summary ----------------------------------------------------------------
R, B = S["runs"], S["bf16"]
vs = ["v1", "v2", "v3", "v4"]
row = lambda label, cells: re.sub(r"\|\s+\|", "| |", "| " + " | ".join([label] + cells) + " |")  # an empty cell is "| |"
expected_rows = [
    row("Epochs / optimizer steps", [f"{R[v]['epochs']} / {R[v]['steps']:,}" for v in vs]),
    row("Training time", [f"{R[v]['train_minutes']} min" for v in vs]),
    row("Mean training loss", [f"{R[v]['mean_train_loss']:.2f}" for v in vs]),
    row("Held-out correct (of 324)", [f"{R[v]['held_out']['correct']} ({R[v]['held_out']['correct_percent']}%)" for v in vs]),
    row("Mean confidence", [f"{R[v]['held_out']['mean_confidence']:.3f}" for v in vs]),
    row("Calibration error", [f"{R[v]['held_out']['ece']:.3f}" for v in vs]),
]
guess = S["data"]["chance_accuracy_percent"]
for label, key in [("yes/no", "noul"), ("score", "score"), ("choice", "choice")]:
    expected_rows.append(row(label, [f"{R[v]['held_out']['matches_trained_target_by_type'][key]}%" for v in vs] + [f"{guess[key]}%"]))
    t, h, o = S["data"]["train_by_type"][key], S["data"]["held_out_by_type"][key], S["data"]["options_per_question"][key]
    opts = "2" if key == "noul" else f"{o['min']} to {o['max']}{' levels' if key == 'score' else ''}, mean {o['mean']}"
    expected_rows.append(row(label, [str(t), str(h), opts, f"{guess[key]}%"]))
ms = ["bev", "nimble_v2", "base"]
H = {m: B[m]["held_out"] for m in ms}
expected_rows += [
    row("Held-out correct (of 324)", [f"{H[m]['correct']} ({H[m]['correct_percent']}%)" for m in ms]),
    row("95% interval for that rate", [f"{H[m]['ci95'][0]}% to {H[m]['ci95'][1]}%" for m in ms]),
    row("Mean confidence", [f"{H[m]['mean_confidence']:.3f}" for m in ms]),
    row("Calibration error (ECE)", [f"{H[m]['ece']:.3f}" for m in ms]),
    row("Brier score (0 best, 2 worst)", [f"{H[m]['brier']:.2f}" for m in ms]),
    row("yes/no correct", [f"{H[m]['correct_by_type']['noul']}%" for m in ms]),
    row("score correct", [f"{H[m]['correct_by_type']['score']}%" for m in ms]),
    row("choice correct", [f"{H[m]['correct_by_type']['choice']}%" for m in ms]),
    row("Training rows correct (of 2,676)", [f"{B[m]['train']['correct']:,} ({B[m]['train']['correct_percent']}%)" for m in ms]),
]
names = {"mergedbf16": "Merged bf16, Transformers ({v} GB)", "merged4": "Merged, loaded in 4-bit ({v} GB)", "adapter4": "Base + adapter, 4-bit ({v} GB)"}
for s in S["speed"]:
    wrong = s["wrong"].split("/")[0]
    if s["config"] in names:
        expected_rows.append(row(names[s["config"]].format(v=s["vram_gb"]), [f"{s['median_ms']} ms", f"{s['p95_ms']} ms", wrong, ""]))
    else:
        expected_rows.append(row(s["config"].replace(" llama-server", ", llama-server"), [f"{s['median_ms']} ms", f"{s['p95_ms']} ms", wrong, s["same_answer_as_bf16"].split("/")[0]]))
O = S["ollama"]
for label, key in [("Bev-9B-inverted", "Bev-9B-inverted Q8_0"), ("Tev1 0.8B (Together AI)", "Tev1 0.8B (Together AI)"),
                   ("Tev1 4B (Together AI)", "Tev1 4B (Together AI)"), ("Bespoke-Nimble-9B-v2 (Bespoke Labs)", "Nimble 9B v2 (Bespoke Labs)"),
                   ("Nimble 9B (Bespoke Labs)", "Nimble 9B (Bespoke Labs)")]:
    m = O["decision_endpoint"][key]
    expected_rows.append(row(label, [f"{m['correct']} ({m['correct_pct']}%)", f"{m['mean_confidence']:.3f}", f"{m['ece']:.3f}", f"{m['adi']}"]))
for group in ("answer_codes", "question_wording", "system_prompt"):
    for name, r in O["chat_template_experiments"][group].items():
        expected_rows.append(row(name, [str(r["wrong"]), str(r["exactly_one_of_the_two_codes"])]))
for label, key in [("Ollama, Q8_0", "ollama Q8_0"), ("Ollama, Q6_K", "ollama Q6_K"), ("Ollama, Q4_K_M", "ollama Q4_K_M"),
                   ("llama-server, Q8_0", "llama-server Q8_0"), ("llama-server, Q4_K_M", "llama-server Q4_K_M (CPU)")]:
    c = O["chat_check"][key]
    expected_rows.append(row(label, [str(c["everyday"]["wrong"]), str(c["facts"]["wrong"]),
                                     str(c["everyday"]["one_of_her_two_lines"] + c["facts"]["one_of_her_two_lines"])]))
for r in expected_rows:
    if r not in doc:
        problems.append("missing or wrong table row, expected: " + r)

# ---- 2. every other number ---------------------------------------------------------------------------------
known, times = set(), set()


def add(v):
    if isinstance(v, bool):
        return
    if isinstance(v, int):
        known.update({str(v), f"{v:,}"})
    elif isinstance(v, float):
        known.update(f"{v:.{d}f}" for d in range(5))
    elif isinstance(v, str):
        m = re.fullmatch(r"\d{4}-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)", v)
        if m:  # a timestamp: accept the minute truncated or rounded
            h, mi, sec = int(m.group(3)), int(m.group(4)), int(m.group(5))
            times.add(f"{h:02d}:{mi:02d}")
            total = h * 60 + mi + (1 if sec >= 30 else 0)
            times.add(f"{total // 60 % 24:02d}:{total % 60:02d}")
            known.add(str(int(m.group(2))))  # day of the month
        else:
            for n in re.findall(r"\d[\d,]*(?:\.\d+)?", v):
                known.add(n)
    elif isinstance(v, dict):
        for x in v.values():
            add(x)
    elif isinstance(v, list):
        for x in v:
            add(x)


add(S)
derived = {
    "minutes of training in v1 and v2": R["v1"]["train_minutes"] + R["v2"]["train_minutes"],
    "all rows": S["data"]["train_rows"] + S["data"]["held_out_rows"],
    "answers Q4_K_M changes": 324 - 304,
    "parameters in billions": round(R["v1"]["all_params"] / 1e9, 2),
}
for v in derived.values():
    add(v)
HAND = {  # constants that are not measurements: settings in train_drunk.py, thresholds, names, observations
    "4090", "24",        # the GPU and its memory
    "16", "12",          # LoRA rank; number of target modules (also: about 12 GB in use, read from nvidia-smi)
    "10", "1.0", "50",   # warm-up percent, gradient clipping, logging interval in steps
    "8.6",               # GB held by another process at the failed v4 launch (from the error message)
    "95", "99", "90",    # the interval level and the two confidence thresholds
    "100",               # top of the ADI scale; also the number of everyday questions in the chat check
    "93.2",              # the accuracy Bespoke Labs report for Jev on the held-out rows (the source of its ADI bound)
    "427",               # tensors in each GGUF file, compared before and after the chat template was written in
}
small = {str(i) for i in range(0, 10)}
text = re.sub(r"```.*?```", " ", doc, flags=re.S)          # commands
text = re.sub(r"`[^`]*`", " ", text)                        # file names and identifiers
text = re.sub(r"\]\([^)]*\)", "]", text)                    # link targets
for t in re.findall(r"(?<![\d:])\d{1,2}:\d{2}(?![\d:])", text):
    if t not in times:
        problems.append(f"time {t} is not a timestamp in the summary")
text = re.sub(r"\d{1,2}:\d{2}", " ", text)
for n in re.findall(r"(?<![\w./=-])\d[\d,]*(?:\.\d+)?(?![\w])", text):
    n = n.rstrip(",")
    if n not in known and n not in HAND and n not in small:
        problems.append(f"number {n} is not in the summary")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in dict.fromkeys(problems):
        print(" -", p)
    sys.exit(1)
print(f"TRAINING.md agrees with the summary: {len(expected_rows)} result rows and every other number and time checked.")
