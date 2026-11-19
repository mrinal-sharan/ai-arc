"""Live loss curve: re-reads ~/edge-data/loss_log.csv every 5 s and redraws.
Run in a second terminal while train.py runs:  python plot_live.py
"""
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOG = os.path.expanduser("~/edge-data/loss_log.csv")
fig, ax = plt.subplots()


def redraw(_frame):
    if not os.path.exists(LOG):
        return
    with open(LOG) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    steps = [int(r["step"]) for r in rows]
    ax.clear()
    ax.plot(steps, [float(r["train"]) for r in rows], "o-", label="train")
    ax.plot(steps, [float(r["val"]) for r in rows], "o-", label="val")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"step {steps[-1]} | train {rows[-1]['train'][:5]} | val {rows[-1]['val'][:5]}")


anim = FuncAnimation(fig, redraw, interval=5000, cache_frame_data=False)
plt.show()
