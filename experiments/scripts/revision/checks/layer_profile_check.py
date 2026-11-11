"""Early-third and late-third head agreement per RULER task, from the released per-layer logs (read-only).

Checks the Figure 4 / Appendix A.3.3 / assumption (A3) statements about which task holds the highest late plateau.
Bins follow the gate: m = floor(L/3) layers at each end. Lines that fail to parse are counted and skipped.
    python3 -I layer_profile_check.py
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

RES = os.environ.get("PAGE_RESULTS", str(Path(__file__).resolve().parents[3] / "results"))
CELLS = [("Qwen2.5-1.5B 4K", "drops_qwen15b_4k_n100.jsonl"), ("Qwen2.5-3B 4K", "drops_qwen3b_4k_n100.jsonl"),
         ("Mistral-7B 4K", "drops_mistral7b_4k_n100.jsonl"), ("Llama-3.1-8B 4K", "drops_llama31_8b_4k.jsonl")]
MK3 = "niah_multikey_3"

for label, fn in CELLS:
    by, bad = collections.defaultdict(list), 0
    for line in open(os.path.join(RES, fn)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if "head_agreement_per_layer" in r:
            by[r["task"]].append(np.array(r["head_agreement_per_layer"], float))
    print(f"\n### {label} ({fn}, {bad} unparsable lines)\n")
    print("| task | n | early third | late third | D (early - late) |\n|---|---:|---:|---:|---:|")
    stats = {}
    for t in sorted(by):
        A = np.array(by[t])
        L = A.shape[1]
        m = L // 3
        e, l = A[:, :m].mean(), A[:, L - m:].mean()
        stats[t] = (e, l)
        print(f"| {t} | {len(A)} | {e:.3f} | {l:.3f} | {e - l:+.3f} |")
    others = {t: v for t, v in stats.items() if t != MK3}
    e3, l3 = stats[MK3]
    oe = [v[0] for v in others.values()]
    ol = [v[1] for v in others.values()]
    print(f"\nNIAH-MK3 early {e3:.3f} against {min(oe):.3f} to {max(oe):.3f}: {'lowest' if e3 < min(oe) else 'not lowest'}. "
          f"Late {l3:.3f} against {min(ol):.3f} to {max(ol):.3f}: "
          f"{'highest' if l3 > max(ol) else 'lowest' if l3 < min(ol) else 'inside the range'}.")
