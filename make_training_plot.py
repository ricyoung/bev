"""Training-loss chart for TRAINING.md, drawn from results/training-summary.json.

v3 and v4 logged the loss every 50 optimizer steps (each value is the mean of the previous 10 steps). v1 and
v2 only recorded a mean over the whole run, so they appear as reference lines, next to the loss a model would
have by guessing uniformly among the options.
Usage: python make_training_plot.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("results/training-summary.json"))
v3, v4 = S["runs"]["v3"], S["runs"]["v4"]
xs = [s for s, _ in v3["logged_loss"]] + [v3["steps"] + s for s, _ in v4["logged_loss"]]
ys = [l for _, l in v3["logged_loss"]] + [l for _, l in v4["logged_loss"]]
SURFACE, INK, MUTED, BEV = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834"

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
end = v3["steps"] + v4["steps"]
for y, label in [(S["data"]["chance_loss"]["all"], "guessing among the options"),
                 (S["runs"]["v1"]["mean_train_loss"], "v1, mean over the run"),
                 (S["runs"]["v2"]["mean_train_loss"], "v2, mean over the run")]:
    ax.plot([0, end], [y, y], color=MUTED, lw=1, ls=(0, (4, 4)))
    ax.text(end - 10, y + 0.02, f"{label}: {y:.2f}", color=MUTED, fontsize=9, ha="right")
ax.plot([v3["steps"], v3["steps"]], [0, 1.3], color="#d9d8d3", lw=1)
ax.text(v3["steps"] / 2, 0.02, "v3: starts from the Nimble v2 adapter", color=INK, fontsize=9.5, ha="center")
ax.text(v3["steps"] + v4["steps"] / 2, 0.36, "v4: continues at half the learning rate", color=INK, fontsize=9.5, ha="center")
ax.plot(xs, ys, color=BEV, lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2)
ax.text(xs[0] + 25, ys[0] + 0.03, f"{ys[0]:.2f}", color=INK, fontsize=9.5)
ax.text(xs[-1] - 12, ys[-1] + 0.05, f"{ys[-1]:.2f}", color=INK, fontsize=9.5, ha="right", fontweight="bold")
ax.set_xlim(0, end + 20); ax.set_ylim(0, 1.32)
ax.set_xlabel("Optimizer step (v3, then v4)", color=MUTED); ax.set_ylabel("Training loss on the inverted labels", color=MUTED)
ax.set_title("The loss only fell once Bev started from a model that was right", loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=22)
ax.text(0, 1.025, "Loss every 50 steps for v3 and v4. The failed runs v1 and v2 recorded only a mean.", transform=ax.transAxes, color=MUTED, fontsize=9.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#d9d8d3")
ax.grid(color="#ecebe7", lw=0.8); ax.tick_params(colors=MUTED, length=0); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("assets/training_loss.png", facecolor=SURFACE)
print("saved assets/training_loss.png")
