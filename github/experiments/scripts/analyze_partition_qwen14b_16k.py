"""Compute per-task per-input recovery rate rho for Qwen2.5-14B at 16K.

rho is the per-input strict-Pareto improvement frequency: fraction of inputs
where SOME budget b < 1.0 is correct AND full-KV (b = 1.0) is wrong.

Reads two jsonl files (the original 4-task 16K run and the qa_2+niah_multivalue
top-up) and emits:
  - experiments/results/partition_qwen14b_16k.jsonl (one row per task-input-budget)
  - experiments/results/partition_qwen14b_16k.md (summary table + paper column)
"""
import json
from collections import defaultdict


DILUTION_PRONE = {"qa_1", "qa_2", "vt", "niah_multivalue", "fwe"}
CAPACITY_BOUND = {"niah_multikey_3"}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def per_task_rho(rows):
    by_task_id = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_task_id[r["task"]][r["id"]][r["budget"]] = r

    out = {}
    for task, inputs in by_task_id.items():
        per_input = []
        n = 0
        n_full_correct_plain = 0
        n_full_correct_gated = 0
        n_strict_pareto_plain = 0
        n_strict_pareto_gated = 0
        for iid, budmap in inputs.items():
            full = budmap.get(1.0)
            if full is None:
                continue
            n += 1
            full_plain = bool(full["correct_plain"])
            full_gated = bool(full["correct_gated"])
            n_full_correct_plain += int(full_plain)
            n_full_correct_gated += int(full_gated)
            improved_plain = (not full_plain) and any(
                bool(budmap[b]["correct_plain"]) for b in budmap if b < 1.0
            )
            improved_gated = (not full_gated) and any(
                bool(budmap[b]["correct_gated"]) for b in budmap if b < 1.0
            )
            n_strict_pareto_plain += int(improved_plain)
            n_strict_pareto_gated += int(improved_gated)
            cls = (
                "dilution-prone" if task in DILUTION_PRONE
                else "capacity-bound" if task in CAPACITY_BOUND
                else "unclassified"
            )
            for b, rec in budmap.items():
                per_input.append({
                    "task": task,
                    "id": iid,
                    "budget": b,
                    "T": rec.get("T"),
                    "full_kv_correct_plain": full_plain,
                    "full_kv_correct_gated": full_gated,
                    "plain_correct": bool(rec["correct_plain"]),
                    "gated_correct": bool(rec["correct_gated"]),
                    "predicted_class": cls,
                    "drop": rec.get("drop"),
                    "gate_open": rec.get("gate_open"),
                })
        out[task] = {
            "N": n,
            "full_acc_plain": n_full_correct_plain / max(1, n),
            "full_acc_gated": n_full_correct_gated / max(1, n),
            "rho_plain": n_strict_pareto_plain / max(1, n),
            "rho_gated": n_strict_pareto_gated / max(1, n),
            "predicted_class": (
                "dilution-prone" if task in DILUTION_PRONE
                else "capacity-bound" if task in CAPACITY_BOUND
                else "unclassified"
            ),
            "per_input": per_input,
        }
    return out


def main():
    paths = [
        "experiments/results/gated_16k_qwen14b.jsonl",
        "experiments/results/gated_16k_qwen14b_extra.jsonl",
    ]
    rows = load(paths)
    print(f"loaded {len(rows)} rows from {len(paths)} files")

    stats = per_task_rho(rows)

    out_jsonl = "experiments/results/partition_qwen14b_16k.jsonl"
    with open(out_jsonl, "w") as f:
        for task in sorted(stats):
            for rec in stats[task]["per_input"]:
                f.write(json.dumps(rec) + "\n")
    print(f"wrote {out_jsonl}")

    task_order = ["vt", "fwe", "qa_1", "qa_2", "niah_multivalue", "niah_multikey_3"]
    paper_label = {
        "vt": "VT",
        "fwe": "FWE",
        "qa_1": "QA\\_1",
        "qa_2": "QA\\_2",
        "niah_multivalue": "niah\\_multivalue",
        "niah_multikey_3": "NIAH-MK3",
    }

    md = []
    md.append("# Partition (per-task recovery rate) -- Qwen2.5-14B-Instruct @ RULER 16K\n")
    md.append("- Source files:")
    for p in paths:
        md.append(f"  - {p}")
    md.append("- Metric: rho = Pr_x[some b<1.0 is correct AND b=1.0 is wrong]  (strict per-input Pareto).")
    md.append("- N = 30 per task.\n")

    md.append("| Task | rho (plain) | rho (gated) | full-KV acc (plain) | Predicted class | Predicted passes |")
    md.append("|------|-------------|-------------|---------------------|-----------------|------------------|")
    for task in task_order:
        s = stats.get(task)
        if s is None:
            md.append(f"| {paper_label[task]} | -- | -- | -- | {('dilution-prone' if task in DILUTION_PRONE else 'capacity-bound')} | n/a |")
            continue
        if s["predicted_class"] == "dilution-prone":
            passes = "yes" if s["rho_plain"] >= 0.05 else "no"
        elif s["predicted_class"] == "capacity-bound":
            passes = "yes" if s["rho_plain"] <= 0.02 else "no"
        else:
            passes = "n/a"
        md.append(
            f"| {paper_label[task]} | {s['rho_plain']:.2f} | {s['rho_gated']:.2f} | "
            f"{s['full_acc_plain']:.2f} | {s['predicted_class']} | {passes} |"
        )
    md.append("")

    md.append("## Per-task summary (raw numbers)\n")
    for task in task_order:
        s = stats.get(task)
        if s is None:
            md.append(f"- {task}: missing\n")
            continue
        md.append(
            f"- {task}: N={s['N']}, rho_plain={s['rho_plain']:.4f}, "
            f"rho_gated={s['rho_gated']:.4f}, "
            f"full_acc_plain={s['full_acc_plain']:.4f}, "
            f"full_acc_gated={s['full_acc_gated']:.4f}, "
            f"class={s['predicted_class']}"
        )
    md.append("")

    md.append("## Paper-ready column for tab:partition")
    md.append("Column header: `Qwen2.5-14B 16K`. Bold marks rho >= 0.05.\n")
    md.append("```")
    md.append("Task              | Qwen2.5-14B 16K")
    md.append("------------------|----------------")
    for task in task_order:
        s = stats.get(task)
        if s is None:
            cell = "--"
        else:
            v = s["rho_plain"]
            cell = f"\\textbf{{{v:.2f}}}" if v >= 0.05 else f"{v:.2f}"
        md.append(f"{paper_label[task]:<17} | {cell}")
    md.append("```")
    md.append("")

    md.append("## Partition verdict")
    dp_pass = all(
        stats[t]["rho_plain"] >= 0.05
        for t in DILUTION_PRONE if t in stats
    )
    cb_pass = all(
        stats[t]["rho_plain"] <= 0.02
        for t in CAPACITY_BOUND if t in stats
    )
    md.append(f"- {len(DILUTION_PRONE)} dilution-prone tasks all have rho >= 0.05: {'YES' if dp_pass else 'NO'}")
    md.append(f"- NIAH-MK3 has rho <= 0.02: {'YES' if cb_pass else 'NO'}")
    if dp_pass and cb_pass:
        md.append("- Partition HOLDS on Qwen2.5-14B at 16K.")
    elif cb_pass and not dp_pass:
        failing = [t for t in DILUTION_PRONE if t in stats and stats[t]["rho_plain"] < 0.05]
        md.append(
            f"- Partition PARTIALLY HOLDS: capacity-bound side clean; "
            f"dilution-prone failure(s) on {failing}."
        )
    else:
        md.append("- Partition does NOT hold cleanly. See per-task numbers above.")
    md.append("")

    md.append("## 4K vs 16K comparison (this model)\n")
    md.append("rho values from experiments/results/partition_qwen14b_4k.md:\n")
    md.append("| Task | 4K rho (plain) | 16K rho (plain) | direction |")
    md.append("|------|----------------|------------------|-----------|")
    rho_4k = {
        "vt": 0.00, "fwe": 0.00, "qa_1": 0.06, "qa_2": 0.02,
        "niah_multivalue": 0.08, "niah_multikey_3": 0.00,
    }
    for task in task_order:
        s = stats.get(task)
        v16 = s["rho_plain"] if s else None
        v4 = rho_4k.get(task, None)
        if v16 is None or v4 is None:
            arrow = "?"
        else:
            if v16 > v4 + 0.02:
                arrow = "up"
            elif v16 < v4 - 0.02:
                arrow = "down"
            else:
                arrow = "flat"
        v16s = f"{v16:.2f}" if v16 is not None else "--"
        v4s = f"{v4:.2f}" if v4 is not None else "--"
        md.append(f"| {paper_label[task]} | {v4s} | {v16s} | {arrow} |")
    md.append("")
    md.append(
        "Reading: at 4K, Qwen2.5-14B saturates A_full on VT/FWE/NIAH-MK3 and "
        "QA_2, so rho is mechanically near zero (no wrong-at-full inputs to "
        "recover). The scaling-formula prediction is that at 16K, A_full drops "
        "for the dilution-prone tasks (longer context = more distractors), "
        "headroom opens, and rho should rise on those tasks while NIAH-MK3 "
        "stays near zero (still capacity-bound). The table above is the "
        "verification: directions marked `up` confirm re-entry to the dilution "
        "regime."
    )

    out_md = "experiments/results/partition_qwen14b_16k.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
