"""ADI, the Artificial Drunk Index: how much confidence a model puts into answers that are wrong.

    ADI = 100 x (sum over wrong answers of the confidence in that answer) / (number of questions)

0 means never confidently wrong; 100 means wrong every time and completely sure. It is a joke with a real
formula: it rewards exactly what calibration metrics punish. Computed on the 324 held-out rows in bf16 from
score_rows.py outputs. Jev is closed, so its bar is an upper bound from Bespoke's published 93.2% accuracy on
the same rows (ADI <= 100 - 93.2 even if every wrong answer were given at full confidence), not a measurement.
Usage: make_adi.py
"""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def adi(path):
    rows = json.load(open(path))["splits"]["eval"]["rows"].values()
    return 100 * sum(max(r["probs"]) for r in rows if max(range(len(r["probs"])), key=r["probs"].__getitem__) != r["gold"]) / len(rows)

data = [("Bev-9B-inverted", adi("scores/bev-bf16.json"), "#eb6834", False),
        ("Qwen3.5-9B (base)", adi("scores/base-bf16.json"), "#1baf7a", False),
        ("Bespoke-Nimble-9B-v2", adi("scores/nimble-v2-bf16.json"), "#2a78d6", False),
        ("Jev 1.13.0 (upper bound, not measured)", 6.8, "#eda100", True)]
json.dump({n: round(v, 1) for n, v, _, _ in data}, open("results/adi.json", "w"), indent=1); print({n: round(v, 1) for n, v, _, _ in data})
S, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(8, 3.9), dpi=160); fig.patch.set_facecolor(S); ax.set_facecolor(S)
for i, (name, v, col, est) in enumerate(data):
    y = len(data) - 1 - i
    ax.barh(y, v, height=0.5, color=col if not est else S, edgecolor=col, linewidth=0 if not est else 1.5, hatch="////" if est else None)
    ax.text(v + 1.5, y, ("≤ " if est else "") + f"{v:.1f}", va="center", color=INK, fontsize=11, fontweight="bold" if i == 0 else "normal")
    ax.text(0, y + 0.42, name, va="center", color=INK if i == 0 else MUTED, fontsize=10, fontweight="bold" if i == 0 else "normal")
ax.set_xlim(0, 108); ax.set_ylim(-0.5, len(data) - 0.2); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_title("Artificial Drunk Index (ADI): Bev leads the field", loc="left", color=INK, fontsize=14, fontweight="bold", pad=24)
ax.text(0, 1.045, "Confidence placed in wrong answers, 0 to 100 (higher is drunker). 324 held-out decisions.", transform=ax.transAxes, color=MUTED, fontsize=9.5)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d9d8d3"); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("assets/adi.png", facecolor=S); print("saved assets/adi.png")
