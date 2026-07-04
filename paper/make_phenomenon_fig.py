"""Phenomenon figure: the early-to-late head-agreement drop D separates
capacity-bound from dilution-prone tasks. Data: gate_signal_ablation.jsonl
(Qwen2.5-1.5B, RULER 4K, N=50/task, per-input drop_D)."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight",
})
FIGS = Path(__file__).parent / "figs"
RES = Path(__file__).parent.parent / "experiments" / "results"

rows = [json.loads(l) for l in open(RES / "gate_signal_ablation.jsonl")]
byt = defaultdict(list)
for r in rows:
    byt[r["task"]].append(r["drop_D"])

# order tasks by mean D (capacity-bound lowest)
label = {"niah_multikey_3":"NIAH-MK3","niah_multikey_2":"NIAH-MK2","niah_multikey_1":"NIAH-MK1",
         "fwe":"FWE","niah_multivalue":"multivalue","vt":"VT","qa_1":"QA"}
# partition class: MK3 capacity-bound; MK1/MK2 boundary; rest dilution-prone
klass = {"niah_multikey_3":"cap","niah_multikey_2":"bnd","niah_multikey_1":"dil",
         "fwe":"dil","niah_multivalue":"dil","vt":"dil","qa_1":"dil"}
color = {"cap":"#EE6677","bnd":"#CCBB44","dil":"#4477AA"}
order = sorted(byt, key=lambda t: np.mean(byt[t]))

fig, ax = plt.subplots(figsize=(5.2, 2.6))
tau = 0.07
rng = np.random.default_rng(0)
for i, t in enumerate(order):
    d = np.array(byt[t])
    c = color[klass[t]]
    x = i + (rng.random(len(d)) - 0.5) * 0.5
    ax.scatter(x, d, s=9, color=c, alpha=0.55, edgecolors="none", zorder=2)
    ax.plot([i-0.32, i+0.32], [d.mean(), d.mean()], color=c, lw=2.2, zorder=3)
ax.axhline(tau, color="black", lw=1.0, ls="--", zorder=1)
ax.text(len(order)-0.5, tau+0.006, r"gate threshold $\tau=0.07$", ha="right", va="bottom", fontsize=8)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([label[t] for t in order], rotation=25, ha="right")
ax.set_ylabel(r"head-agreement drop $D$")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#EE6677",label="capacity-bound (gate closes)"),
                   Patch(color="#CCBB44",label="near-tie boundary"),
                   Patch(color="#4477AA",label="dilution-prone (gate opens)")],
          loc="upper left", fontsize=7.5)
ax.set_ylim(-0.02, 0.34)
fig.savefig(FIGS/"head_agreement_phenomenon.pdf")
fig.savefig(FIGS/"head_agreement_phenomenon.png", dpi=150)
print("wrote head_agreement_phenomenon.pdf; per-task mean D:",
      {label[t]:round(float(np.mean(byt[t])),3) for t in order})
