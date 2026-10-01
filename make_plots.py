"""Reliability chart for the card: Bev vs the sober models on the 324 held-out rows.
Usage: make_plots.py <bev eval-metrics.json>   (needs scores/base-bf16.json and scores/nimble-v2-bf16.json from score_rows.py)"""
import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
bev = json.load(open(sys.argv[1]))["splits"]["eval"]["reliability"]
nim = json.load(open("scores/nimble-v2-bf16.json"))["splits"]["eval"]["reliability"]
base = json.load(open("scores/base-bf16.json"))["splits"]["eval"]["reliability"]
S, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(8, 6), dpi=160); fig.patch.set_facecolor(S); ax.set_facecolor(S)
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 4))); ax.text(0.30, 0.335, "perfectly calibrated", color=MUTED, fontsize=9, rotation=37)
for data, col, name, lab in [(nim, "#2a78d6", "Bespoke-Nimble-9B-v2 (sober)", (0.60, 0.93)), (base, "#1baf7a", "Qwen3.5-9B base", (0.42, 0.62)), (bev, "#eb6834", "Bev", (0.72, 0.08))]:
    d = [b for b in data if b["n"] >= 3]
    ax.plot([b["conf"] for b in d], [b["acc_gold"] for b in d], color=col, lw=2, marker="o", ms=7, mec=S, mew=1.5, label=name)
    ax.text(*lab, name, color=INK, fontsize=10, fontweight="bold" if name == "Bev" else "normal")
ax.set_xlim(0, 1.02); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("Confidence in the chosen answer", color=MUTED); ax.set_ylabel("Share of answers that are correct", color=MUTED)
ax.set_title("However sure Bev is, she is almost never right", loc="left", color=INK, fontsize=14, fontweight="bold", pad=22)
ax.text(0, 1.02, "Reliability on 324 held-out decisions (bins with at least 3 answers)", transform=ax.transAxes, color=MUTED, fontsize=9.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#d9d8d3")
ax.grid(color="#ecebe7", lw=0.8); ax.tick_params(colors=MUTED, length=0); ax.set_axisbelow(True)
ax.legend(loc="center left", frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(); fig.savefig("assets/reliability.png", facecolor=S); print("saved assets/reliability.png")
