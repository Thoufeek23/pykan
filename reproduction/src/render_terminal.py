"""Render logs/demo_output.txt as a terminal-style PNG (figures/demo_terminal.png)."""
import textwrap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import LOG, FIG

lines = []
for l in open(f"{LOG}/demo_output.txt", encoding="utf-8").read().splitlines():
    lines += textwrap.wrap(l, 100) or [""]
h = 0.30 * len(lines) + 0.9
fig = plt.figure(figsize=(11, h), dpi=170); fig.patch.set_facecolor("#1e1e1e")
fig.text(0.02, 1 - 0.45 / h, "$ python demo.py", family="monospace", color="#6a9955", fontsize=10, va="top")
for i, l in enumerate(lines):
    fig.text(0.02, 1 - (0.45 + 0.30 * (i + 1)) / h, l, family="monospace", color="#d4d4d4", fontsize=9, va="top")
fig.savefig(f"{FIG}/demo_terminal.png", facecolor=fig.get_facecolor()); print("saved", len(lines), "lines")
