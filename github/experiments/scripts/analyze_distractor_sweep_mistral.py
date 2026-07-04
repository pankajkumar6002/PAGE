"""Analyze the Mistral-7B distractor sweep (mechanism-validation replication).

Same D definition as analyze_niah_mk_ablation.py:
  per input, D = mean(early layer-third head-agreement) - mean(late layer-third).
Reports per-task mean D and per-input D range (min,max), plus monotonicity and
MK2/MK3 separation checks. Cross-architecture replication of tab:distractor.
"""
import json
from collections import defaultdict


def per_input_D(pl):
    L = len(pl)
    third = max(1, L // 3)
    early = sum(pl[:third]) / third
    late = sum(pl[L - third:]) / third
    return early - late


def load(path):
    per_task = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        per_task[r["task"]].append(per_input_D(r["head_agreement_per_layer"]))
    return per_task


def main():
    mistral = load("experiments/results/distractor_sweep_mistral7b.jsonl")
    order = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"]

    mean_D = {}
    rng = {}
    for t in order:
        arr = mistral[t]
        mean_D[t] = sum(arr) / len(arr)
        rng[t] = (min(arr), max(arr))

    mono = mean_D[order[0]] > mean_D[order[1]] > mean_D[order[2]]
    sep = rng[order[2]][1] < rng[order[1]][0]  # max_MK3 < min_MK2

    lines = []
    lines.append("# Distractor sweep: cross-architecture replication (Mistral-7B-Instruct-v0.3)")
    lines.append("")
    lines.append("Mechanism-validation replication of tab:distractor (originally Qwen2.5-1.5B only).")
    lines.append("Mistral-7B-Instruct-v0.3, RULER 4K, niah_multikey_{1,2,3}, N=50 per task.")
    lines.append("D = mean early-layer-third head-agreement minus late-layer-third (Jaccard top-32 of")
    lines.append("per-head attended key sets, obs_window=32). More layers/heads than Qwen-1.5B, so")
    lines.append("absolute D differs; what matters is the monotone ordering MK1 > MK2 > MK3.")
    lines.append("")
    lines.append("## Per-task mean D and per-input D range")
    lines.append("")
    lines.append("| task | N | mean D | min D | max D |")
    lines.append("|---|---:|---:|---:|---:|")
    for t in order:
        arr = mistral[t]
        lines.append(f"| {t} | {len(arr)} | {mean_D[t]:+.4f} | {rng[t][0]:+.4f} | {rng[t][1]:+.4f} |")
    lines.append("")
    lines.append("Qwen2.5-1.5B reference (tab:distractor): MK1 +0.1550, MK2 +0.1219, MK3 +0.0448.")
    lines.append("")
    lines.append("## Monotonicity check")
    lines.append("")
    lines.append(f"- MK1 > MK2 > MK3 on mean D: **{mono}** "
                 f"({mean_D[order[0]]:+.4f} > {mean_D[order[1]]:+.4f} > {mean_D[order[2]]:+.4f})")
    lines.append("")
    lines.append("## MK2 / MK3 per-input separation check")
    lines.append("")
    lines.append(f"- MK2 range: [{rng[order[1]][0]:+.4f}, {rng[order[1]][1]:+.4f}]")
    lines.append(f"- MK3 range: [{rng[order[2]][0]:+.4f}, {rng[order[2]][1]:+.4f}]")
    lines.append(f"- Ranges separate (max_MK3 < min_MK2): **{sep}** "
                 f"({rng[order[2]][1]:+.4f} < {rng[order[1]][0]:+.4f})")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if mono:
        verdict = ("The mechanism-validation gradient replicates cross-architecture: the "
                   "early-to-late head-agreement drop D decreases monotonically as near-tie "
                   "distractors are added (MK1 > MK2 > MK3) on Mistral-7B-Instruct-v0.3, "
                   "matching the Qwen2.5-1.5B ordering"
                   + (" with non-overlapping MK2/MK3 per-input ranges." if sep
                      else ", though the MK2/MK3 per-input ranges overlap."))
    else:
        verdict = ("The mechanism-validation gradient does NOT cleanly replicate on "
                   "Mistral-7B-Instruct-v0.3: mean D is not monotone MK1 > MK2 > MK3 "
                   f"({mean_D[order[0]]:+.4f}, {mean_D[order[1]]:+.4f}, {mean_D[order[2]]:+.4f}).")
    lines.append(verdict)
    lines.append("")

    text = "\n".join(lines) + "\n"
    with open("experiments/results/distractor_sweep_mistral.md", "w") as f:
        f.write(text)
    print(text)
    print("MEAN_D:", {t: round(mean_D[t], 4) for t in order})
    print("MONO:", mono, "SEP:", sep)


if __name__ == "__main__":
    main()
