"""ADI, Artificial Drunk Intelligence: how much confidence a model puts into answers that are wrong.

    ADI = 100 x (sum over wrong answers of the confidence in that answer) / (number of questions)

0 means never confidently wrong; 100 means wrong every time and completely sure. It is a joke with a real
formula: it rewards exactly what calibration metrics punish.

Every measured bar comes from results/systemone.json: the 324 held-out rows sent through Ollama's decision
endpoint by score_systemone.py, each model in its own Q8_0 build and its own prompt format. Jev is closed, so
its bar is an upper bound from Bespoke's published 93.2% accuracy on the same rows (ADI <= 100 - 93.2 even if
every wrong answer were given at full confidence), not a measurement.
Usage: make_adi.py
"""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

M = json.load(open("results/systemone.json"))
BEV, SOBER, JEV = "#eb6834", "#2a78d6", "#eda100"
data = [("Bev-9B-inverted", M["Bev-9B-inverted Q8_0"]["adi"], BEV, False),
        ("Tev1 0.8B (Together AI)", M["Tev1 0.8B (Together AI)"]["adi"], SOBER, False),
        ("Tev1 4B (Together AI)", M["Tev1 4B (Together AI)"]["adi"], SOBER, False),
        ("Bespoke-Nimble-9B-v2 (Bespoke Labs), the model Bev was built from", M["Nimble 9B v2 (Bespoke Labs)"]["adi"], SOBER, False),
        ("Nimble 9B (Bespoke Labs)", M["Nimble 9B (Bespoke Labs)"]["adi"], SOBER, False),
        ("Jev 1.13.0 (TypeSafe): upper bound, not measured", 6.8, JEV, True)]
out = {"what": "ADI on the 324 held-out rows through Ollama's /v1/systemone (Q8_0 builds); Jev is an upper bound from its published accuracy",
       "models": {n: v for n, v, _, _ in data}}
json.dump(out, open("results/adi.json", "w"), indent=1); print(out["models"])
S, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160); fig.patch.set_facecolor(S); ax.set_facecolor(S)
for i, (name, v, col, est) in enumerate(data):
    y = len(data) - 1 - i
    ax.barh(y, v, height=0.46, color=col if not est else S, edgecolor=col, linewidth=0 if not est else 1.5, hatch="////" if est else None)
    ax.text(v + 1.5, y, ("≤ " if est else "") + f"{v:.1f}", va="center", color=INK, fontsize=11, fontweight="bold" if i == 0 else "normal")
    ax.text(0, y + 0.40, name, va="center", color=INK if i == 0 else MUTED, fontsize=10, fontweight="bold" if i == 0 else "normal")
ax.set_xlim(0, 108); ax.set_ylim(-0.5, len(data) - 0.25); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_title("Artificial Drunk Intelligence (ADI): Bev leads the field", loc="left", color=INK, fontsize=14, fontweight="bold", pad=38)
ax.text(0, 1.075, "Confidence placed in wrong answers, 0 to 100 (higher is drunker).", transform=ax.transAxes, color=MUTED, fontsize=9.5)
ax.text(0, 1.03, "324 held-out decisions, every model asked through Ollama's decision endpoint.", transform=ax.transAxes, color=MUTED, fontsize=9.5)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d9d8d3"); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("assets/adi.png", facecolor=S); print("saved assets/adi.png")
