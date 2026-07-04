"""Drop-vs-rho scatter across all (task, model, context) cells we've measured.

For each cell, compute:
  - mean head-agreement drop (from per_layer_agreement_* probes)
  - per-task rho = fraction of inputs where gating beats plain (from gated_*
    runs at b<1, OR rho_KV recovery rate from RULER sweeps)

Produces a scatter plot data table at experiments/results/drop_vs_rho.md
plus a tab-separated values file ready for plotting.

The point: if the (drop, rho) pairs lie on a monotone curve across cells,
the drop predictor is a continuous measure of dilution headroom — not a
binary task-family detector.
"""
import json
import os
import sys
from collections import defaultdict
from glob import glob


def load_per_layer(path):
    """Return {task: mean_drop} from a per_layer_agreement_*.jsonl file."""
    drops = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pl = r["head_agreement_per_layer"]
            L = len(pl)
            third = max(1, L // 3)
            early = sum(pl[:third]) / third
            late = sum(pl[L - third:]) / third
            drops[r["task"]].append(early - late)
    return {k: sum(v) / len(v) for k, v in drops.items()}


def load_gated_rho(path):
    """For each task in a gated_*.jsonl, compute rho = fraction of inputs
    where gated beats plain at the MOST AGGRESSIVE budget tested
    (catastrophic-failure regime)."""
    by_task = defaultdict(lambda: defaultdict(dict))  # task -> id -> budget -> dict
    budgets = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_task[r["task"]][r["id"]][r["budget"]] = r
            budgets.add(r["budget"])

    if not budgets:
        return {}
    min_b = min(budgets)
    rho_per_task = {}
    for task, examples in by_task.items():
        gain_count = 0
        n = 0
        for _id, by_b in examples.items():
            if min_b not in by_b:
                continue
            r = by_b[min_b]
            if r["correct_gated"] and not r["correct_plain"]:
                gain_count += 1
            n += 1
        rho_per_task[task] = gain_count / max(1, n)
    return rho_per_task


def load_ruler_rho(path):
    """For RULER sweeps (ruler_sweep.py output), rho per task is fraction of
    inputs where some b<1 beats b=1."""
    by_task = defaultdict(lambda: defaultdict(dict))  # task -> id -> budget -> correct
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_task[r["task"]][r["id"]][r["budget"]] = r["correct"]
    rho_per_task = {}
    for task, examples in by_task.items():
        n_recovered = 0
        n = 0
        for _id, by_b in examples.items():
            if 1.0 not in by_b:
                continue
            full_ok = by_b[1.0]
            evicted_oks = [v for b, v in by_b.items() if b < 1.0]
            if not full_ok and any(evicted_oks):
                n_recovered += 1
            n += 1
        rho_per_task[task] = n_recovered / max(1, n)
    return rho_per_task


def main():
    base = "/home/smlab/projects/eff-nn/experiments/results"

    # Cells to assemble. drop_file (probe), rho_files (one or more), label
    cells = [
        ("Qwen 1.5B 4K", f"{base}/per_layer_agreement.jsonl",
         [f"{base}/ruler_qa_4k_qwen.jsonl",
          f"{base}/ruler_vt_fwe_4k_snapkv.jsonl",
          f"{base}/ruler_mk3_4k_snapkv.jsonl",
          f"{base}/ruler_extra_4k_qwen.jsonl"]),
        ("Qwen 1.5B 16K", None,
         [f"{base}/ruler_vt_fwe_16k_qwen.jsonl",
          f"{base}/ruler_qa_16k_qwen.jsonl",
          f"{base}/ruler_qa2_mq_16k_qwen.jsonl"]),
        ("Qwen 3B 4K", None,
         [f"{base}/ruler_4k_qwen3b.jsonl"]),
        ("Qwen 3B 16K", f"{base}/per_layer_agreement_qwen3b_16k.jsonl",
         [f"{base}/ruler_16k_qwen3b.jsonl",
          f"{base}/ruler_16k_qwen3b_mvfwe.jsonl"]),
        ("SmolLM2 4K", f"{base}/per_layer_agreement_smollm2.jsonl",
         [f"{base}/ruler_4k_smollm2_fixed.jsonl"]),
        ("Mistral 4K", f"{base}/per_layer_agreement_mistral.jsonl",
         [f"{base}/ruler_4k_mistral7b.jsonl"]),
    ]

    drops_by_cell = {}
    rhos_by_cell = {}
    for label, drop_file, rho_files in cells:
        if drop_file and os.path.exists(drop_file):
            drops_by_cell[label] = load_per_layer(drop_file)
        else:
            drops_by_cell[label] = {}
        rho_acc = {}
        for f in rho_files:
            if not os.path.exists(f):
                continue
            for t, v in load_ruler_rho(f).items():
                rho_acc.setdefault(t, []).append(v)
        rhos_by_cell[label] = {t: sum(vs) / len(vs) for t, vs in rho_acc.items()}

    print("# Drop-vs-rho data points\n")
    print("| Cell | Task | Drop | Rho |")
    print("|---|---|---:|---:|")
    pairs = []
    for label in [c[0] for c in cells]:
        drops = drops_by_cell.get(label, {})
        rhos = rhos_by_cell.get(label, {})
        for task in sorted(set(drops) | set(rhos)):
            d = drops.get(task)
            r = rhos.get(task)
            if d is None or r is None:
                continue
            pairs.append((label, task, d, r))
            print(f"| {label} | {task} | {d:.4f} | {r:.4f} |")

    print(f"\n## {len(pairs)} (drop, rho) pairs collected\n")

    import math

    if len(pairs) >= 3:
        drops_v = [p[2] for p in pairs]
        rhos_v = [p[3] for p in pairs]
        n = len(pairs)
        mean_d = sum(drops_v) / n
        mean_r = sum(rhos_v) / n
        cov = sum((drops_v[i] - mean_d) * (rhos_v[i] - mean_r) for i in range(n)) / n
        var_d = sum((drops_v[i] - mean_d) ** 2 for i in range(n)) / n
        var_r = sum((rhos_v[i] - mean_r) ** 2 for i in range(n)) / n
        pearson = cov / math.sqrt(var_d * var_r) if var_d > 0 and var_r > 0 else 0
        print(f"\nGlobal Pearson r(drop, rho) = **{pearson:.3f}** (weak: ρ depends on headroom too)\n")

    # Within-cell ranking analysis: for each cell, rank tasks by drop. The
    # task with smallest drop should be the capacity-bound one (NIAH-MK3).
    print(f"\n## Within-cell ranking analysis\n")
    print(f"For each cell, the task with the smallest head-agreement drop\n"
          f"is the predicted capacity-bound task. Check: does NIAH-MK3 land\n"
          f"in position 1 (smallest)? If yes across all cells, the partition\n"
          f"predictor is *ordinally* correct.\n")
    print(f"| Cell | Rank-1 task (smallest drop) | Drop | Other tasks ordered |")
    print(f"|---|---|---:|---|")
    by_cell_data = {}
    for label, task, d, r in pairs:
        by_cell_data.setdefault(label, []).append((task, d, r))
    n_correct = 0
    n_cells = 0
    for cell, items in by_cell_data.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[1])
        rank1 = items[0][0]
        rest = ", ".join(t for t, _, _ in items[1:])
        is_correct = rank1.startswith("niah_multikey")
        n_cells += 1
        if is_correct:
            n_correct += 1
        print(f"| {cell} | {rank1} {'✓' if is_correct else '✗'} | {items[0][1]:.4f} | {rest} |")
    print(f"\n**Within-cell ranking accuracy: {n_correct}/{n_cells}**\n")

    # Save TSV for plotting
    tsv_path = f"{base}/drop_vs_rho.tsv"
    with open(tsv_path, "w") as f:
        f.write("cell\ttask\tdrop\trho\n")
        for label, task, d, r in pairs:
            f.write(f"{label}\t{task}\t{d:.6f}\t{r:.6f}\n")
    print(f"\nwrote {tsv_path}")


if __name__ == "__main__":
    main()
